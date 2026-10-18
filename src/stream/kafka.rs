//! Apache Kafka stream driver implementation using pure-Rust `rskafka`.

use crate::stream::driver::{ConsumedMessage, MessageAcknowledger, StreamDriver};
use crate::types::InferenceError;
use async_trait::async_trait;
use rskafka::client::partition::{OffsetAt, PartitionClient};
use rskafka::client::{Client, ClientBuilder};
use rskafka::record::Record;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

struct KafkaAcknowledger {
    offset: i64,
    current_offset: Arc<AtomicI64>,
}

#[async_trait]
impl MessageAcknowledger for KafkaAcknowledger {
    async fn ack(&self) -> Result<(), InferenceError> {
        let next_offset = self.offset.saturating_add(1);
        self.current_offset
            .fetch_max(next_offset, Ordering::Relaxed);
        Ok(())
    }

    async fn nack(&self, _requeue: bool) -> Result<(), InferenceError> {
        // Offset remains at current position
        Ok(())
    }
}

/// Apache Kafka stream driver utilizing pure-Rust asynchronous `rskafka`.
pub struct KafkaStreamDriver {
    client: Client,
    input_topic: String,
    partition_client: Arc<Mutex<Option<Arc<PartitionClient>>>>,
    offset: Arc<AtomicI64>,
}

impl KafkaStreamDriver {
    /// Connects to a Kafka broker cluster.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if connecting fails.
    pub async fn connect(
        endpoint: &str,
        input_topic: impl Into<String>,
    ) -> Result<Self, InferenceError> {
        let brokers = vec![endpoint.to_string()];
        let client = ClientBuilder::new(brokers).build().await.map_err(|e| {
            InferenceError::Runtime(format!("Failed to connect to Kafka at '{endpoint}': {e}"))
        })?;

        Ok(Self {
            client,
            input_topic: input_topic.into(),
            partition_client: Arc::new(Mutex::new(None)),
            offset: Arc::new(AtomicI64::new(0)),
        })
    }

    async fn get_or_create_partition_client(
        &self,
        topic: &str,
    ) -> Result<Arc<PartitionClient>, InferenceError> {
        let mut guard = self.partition_client.lock().await;
        if let Some(ref pc) = *guard {
            return Ok(pc.clone());
        }

        let pc = self.client.partition_client(topic, 0).await.map_err(|e| {
            InferenceError::Runtime(format!(
                "Failed to get Kafka partition client for '{topic}': {e}"
            ))
        })?;

        let arc_pc = Arc::new(pc);
        *guard = Some(arc_pc.clone());
        Ok(arc_pc)
    }
}

#[async_trait]
impl StreamDriver for KafkaStreamDriver {
    fn broker_name(&self) -> &'static str {
        "kafka"
    }

    async fn ensure_topic_exists(
        &self,
        topic_name: &str,
        _partitions: usize,
    ) -> Result<(), InferenceError> {
        // Attempt to access partition 0 metadata; if not present and auto.create.topics is enabled,
        // broker creates it, or controller handles topic provisioning.
        match self.client.partition_client(topic_name, 0).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("UnknownTopicOrPartition")
                    || err_str.contains("LEADER_NOT_AVAILABLE")
                {
                    // In Kafka with auto.create.topics.enable=true, broker initiates topic creation
                    // Give broker a brief window to settle metadata
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    Ok(())
                } else {
                    Err(InferenceError::Runtime(format!(
                        "Kafka topic '{topic_name}' check failed: {e}"
                    )))
                }
            }
        }
    }

    async fn poll_next(
        &self,
        timeout: Duration,
    ) -> Result<Option<ConsumedMessage>, InferenceError> {
        let pc = match self.get_or_create_partition_client(&self.input_topic).await {
            Ok(pc) => pc,
            Err(_) => return Ok(None),
        };

        let current_offset = self.offset.load(Ordering::Relaxed);
        let max_wait_ms = i32::try_from(timeout.as_millis()).unwrap_or(200);

        // Fetch record with timeout
        let fetch_res = pc
            .fetch_records(current_offset, 1..1_048_576, max_wait_ms)
            .await;

        match fetch_res {
            Ok((records, _high_watermark)) => {
                if let Some(record_and_offset) = records.into_iter().next() {
                    let msg_offset = record_and_offset.offset;
                    let payload = record_and_offset.record.record.value.unwrap_or_default();
                    let message_id = msg_offset.to_string();

                    let mut headers = HashMap::new();
                    for (k, v) in record_and_offset.record.record.headers {
                        if let Ok(val_str) = std::str::from_utf8(&v) {
                            headers.insert(k, val_str.to_string());
                        }
                    }

                    let acknowledger = Arc::new(KafkaAcknowledger {
                        offset: msg_offset,
                        current_offset: self.offset.clone(),
                    });

                    Ok(Some(ConsumedMessage {
                        message_id,
                        topic: self.input_topic.clone(),
                        payload,
                        headers,
                        acknowledger,
                    }))
                } else {
                    Ok(None)
                }
            }
            Err(e) => {
                // If offset is out of range, seek to earliest / latest
                let err_str = e.to_string();
                if err_str.contains("OffsetOutOfRange") {
                    if let Ok(earliest) = pc.get_offset(OffsetAt::Earliest).await {
                        self.offset.store(earliest, Ordering::Relaxed);
                    }
                }
                Ok(None)
            }
        }
    }

    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<(), InferenceError> {
        let pc = self.client.partition_client(topic, 0).await.map_err(|e| {
            InferenceError::Runtime(format!("Failed to get partition client for '{topic}': {e}"))
        })?;

        let record = Record {
            key: None,
            value: Some(payload.to_vec()),
            headers: std::collections::BTreeMap::new(),
            timestamp: rskafka::chrono::Utc::now(),
        };

        pc.produce(
            vec![record],
            rskafka::client::producer::aggregator::RecordAggregator::new(1024),
        )
        .await
        .map_err(|e| {
            InferenceError::Runtime(format!("Failed to produce Kafka message to '{topic}': {e}"))
        })?;

        Ok(())
    }
}
