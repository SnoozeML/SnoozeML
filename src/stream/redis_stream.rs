//! Redis Streams driver implementation using `redis` with consumer groups and `MKSTREAM` auto-creation.

use crate::stream::driver::{ConsumedMessage, MessageAcknowledger, StreamDriver};
use crate::types::InferenceError;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::streams::{StreamId, StreamKey, StreamReadReply};
use redis::{cmd, Client, RedisError};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

struct RedisAcknowledger {
    stream_key: String,
    group_name: String,
    message_id: String,
    conn: Arc<Mutex<ConnectionManager>>,
}

#[async_trait]
impl MessageAcknowledger for RedisAcknowledger {
    async fn ack(&self) -> Result<(), InferenceError> {
        let mut conn = self.conn.lock().await;
        let _: () = cmd("XACK")
            .arg(&self.stream_key)
            .arg(&self.group_name)
            .arg(&self.message_id)
            .query_async(&mut *conn)
            .await
            .map_err(|e| InferenceError::Runtime(format!("Redis XACK failed: {e}")))?;
        Ok(())
    }

    async fn nack(&self, _requeue: bool) -> Result<(), InferenceError> {
        // In Redis Streams, unacknowledged messages remain in the PEL (Pending Entries List)
        // so leaving it unacked allows claim or consumer failover.
        Ok(())
    }
}

/// Redis Streams driver coordinating consumer groups and non-blocking reads.
pub struct RedisStreamDriver {
    stream_key: String,
    group_name: String,
    consumer_name: String,
    conn: Arc<Mutex<ConnectionManager>>,
}

impl RedisStreamDriver {
    /// Connects to Redis and initializes a multiplexed connection manager.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if connecting fails.
    pub async fn connect(
        endpoint: &str,
        stream_key: impl Into<String>,
        group_name: impl Into<String>,
        consumer_name: impl Into<String>,
    ) -> Result<Self, InferenceError> {
        let client = Client::open(endpoint).map_err(|e| {
            InferenceError::Runtime(format!("Failed to parse Redis endpoint '{endpoint}': {e}"))
        })?;

        let conn = client.get_connection_manager().await.map_err(|e| {
            InferenceError::Runtime(format!("Failed to connect to Redis at '{endpoint}': {e}"))
        })?;

        Ok(Self {
            stream_key: stream_key.into(),
            group_name: group_name.into(),
            consumer_name: consumer_name.into(),
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl StreamDriver for RedisStreamDriver {
    fn broker_name(&self) -> &'static str {
        "redis"
    }

    async fn ensure_topic_exists(
        &self,
        topic_name: &str,
        _partitions: usize,
    ) -> Result<(), InferenceError> {
        let mut conn = self.conn.lock().await;

        // XGROUP CREATE <key> <group> $ MKSTREAM
        // MKSTREAM automatically creates the stream if it does not already exist.
        let res: Result<(), RedisError> = cmd("XGROUP")
            .arg("CREATE")
            .arg(topic_name)
            .arg(&self.group_name)
            .arg("$")
            .arg("MKSTREAM")
            .query_async(&mut *conn)
            .await;

        match res {
            Ok(()) => Ok(()),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("BUSYGROUP") || err_str.contains("already exists") {
                    // Consumer group already exists; target stream is active
                    Ok(())
                } else {
                    Err(InferenceError::Runtime(format!(
                        "Failed to create Redis consumer group on '{topic_name}': {e}"
                    )))
                }
            }
        }
    }

    async fn poll_next(
        &self,
        timeout: Duration,
    ) -> Result<Option<ConsumedMessage>, InferenceError> {
        let mut conn = self.conn.lock().await;
        let block_ms = u64::try_from(timeout.as_millis()).unwrap_or(100);

        // XREADGROUP GROUP <group> <consumer> BLOCK <ms> COUNT 1 STREAMS <stream> >
        let reply_res: Result<StreamReadReply, RedisError> = cmd("XREADGROUP")
            .arg("GROUP")
            .arg(&self.group_name)
            .arg(&self.consumer_name)
            .arg("BLOCK")
            .arg(block_ms)
            .arg("COUNT")
            .arg(1)
            .arg("STREAMS")
            .arg(&self.stream_key)
            .arg(">")
            .query_async(&mut *conn)
            .await;

        let reply = match reply_res {
            Ok(r) => r,
            Err(e) => {
                // If it was a nil reply (timeout on empty stream), return Ok(None)
                if e.is_connection_dropped() {
                    return Err(InferenceError::Runtime(format!(
                        "Redis connection dropped: {e}"
                    )));
                }
                return Ok(None);
            }
        };

        let stream_key_obj: &StreamKey = match reply.keys.first() {
            Some(k) => k,
            None => return Ok(None),
        };

        let stream_id: &StreamId = match stream_key_obj.ids.first() {
            Some(id) => id,
            None => return Ok(None),
        };

        let message_id = stream_id.id.clone();

        // Extract payload bytes from field "payload", "data", or first field
        let mut payload = Vec::new();
        let mut headers = HashMap::new();

        for (k, v) in &stream_id.map {
            if k == "payload" || k == "data" {
                if let redis::Value::BulkString(bytes) = v {
                    payload = bytes.clone();
                }
            } else if let redis::Value::BulkString(bytes) = v {
                if let Ok(s) = std::str::from_utf8(bytes) {
                    headers.insert(k.clone(), s.to_string());
                }
            }
        }

        if payload.is_empty() {
            // Fallback: if no dedicated payload field, try first value
            if let Some((_k, redis::Value::BulkString(bytes))) = stream_id.map.iter().next() {
                payload = bytes.clone();
            }
        }

        let acknowledger = Arc::new(RedisAcknowledger {
            stream_key: self.stream_key.clone(),
            group_name: self.group_name.clone(),
            message_id: message_id.clone(),
            conn: self.conn.clone(),
        });

        Ok(Some(ConsumedMessage {
            message_id,
            topic: self.stream_key.clone(),
            payload,
            headers,
            acknowledger,
        }))
    }

    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<(), InferenceError> {
        let mut conn = self.conn.lock().await;

        // XADD <topic> * payload <payload>
        // XADD automatically creates the stream key if it does not already exist
        let _: String = cmd("XADD")
            .arg(topic)
            .arg("*")
            .arg("payload")
            .arg(payload)
            .query_async(&mut *conn)
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!("Failed to XADD message to '{topic}': {e}"))
            })?;

        Ok(())
    }
}
