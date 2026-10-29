//! Apache Kafka stream driver implementation using pure-Rust `rskafka`.

use crate::stream::driver::{ConsumedMessage, MessageAcknowledger, StreamDriver};
use crate::types::InferenceError;
use async_trait::async_trait;
use rskafka::client::partition::{Compression, OffsetAt, PartitionClient, UnknownTopicHandling};
use rskafka::client::{Client, ClientBuilder};
use rskafka::record::{Record, RecordAndOffset};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

struct KafkaAcknowledger {
    offset: i64,
    fetch_offset: Arc<AtomicI64>,
    committed_offset: Arc<AtomicI64>,
    prefetch_buffer: Arc<Mutex<VecDeque<RecordAndOffset>>>,
}

#[async_trait]
impl MessageAcknowledger for KafkaAcknowledger {
    async fn ack(&self) -> Result<(), InferenceError> {
        let next_offset = self.offset.saturating_add(1);
        self.committed_offset
            .fetch_max(next_offset, Ordering::AcqRel);
        Ok(())
    }

    async fn nack(&self, requeue: bool) -> Result<(), InferenceError> {
        if requeue {
            let mut guard = self
                .prefetch_buffer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.clear();
            self.fetch_offset.fetch_min(self.offset, Ordering::AcqRel);
        }
        Ok(())
    }
}

/// Apache Kafka stream driver utilizing pure-Rust asynchronous `rskafka`.
pub struct KafkaStreamDriver {
    client: Client,
    input_topic: String,
    partition_clients: RwLock<HashMap<String, Arc<PartitionClient>>>,
    /// Next record offset to fetch or dispatch from the partition log.
    fetch_offset: Arc<AtomicI64>,
    /// Highest offset acknowledged after inference and egress publication.
    committed_offset: Arc<AtomicI64>,
    /// Local prefetch queue buffering records returned in a single Kafka batch fetch.
    prefetch_buffer: Arc<Mutex<VecDeque<RecordAndOffset>>>,
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
            partition_clients: RwLock::new(HashMap::new()),
            fetch_offset: Arc::new(AtomicI64::new(0)),
            committed_offset: Arc::new(AtomicI64::new(0)),
            prefetch_buffer: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Returns the next offset scheduled to be fetched or dispatched.
    #[must_use]
    pub fn fetch_offset(&self) -> i64 {
        self.fetch_offset.load(Ordering::Acquire)
    }

    /// Returns the highest offset positively acknowledged by downstream workers.
    #[must_use]
    pub fn committed_offset(&self) -> i64 {
        self.committed_offset.load(Ordering::Acquire)
    }

    async fn get_or_create_partition_client(
        &self,
        topic: &str,
    ) -> Result<Arc<PartitionClient>, InferenceError> {
        {
            let guard = self
                .partition_clients
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(pc) = guard.get(topic) {
                return Ok(pc.clone());
            }
        }

        let pc = self
            .client
            .partition_client(topic, 0, UnknownTopicHandling::Retry)
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to get Kafka partition client for '{topic}': {e}"
                ))
            })?;

        let arc_pc = Arc::new(pc);
        {
            let mut guard = self
                .partition_clients
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.insert(topic.to_string(), arc_pc.clone());
        }
        Ok(arc_pc)
    }

    fn build_consumed_message(&self, record_and_offset: RecordAndOffset) -> ConsumedMessage {
        let msg_offset = record_and_offset.offset;
        let next_fetch = msg_offset.saturating_add(1);
        self.fetch_offset.fetch_max(next_fetch, Ordering::AcqRel);

        let payload = record_and_offset.record.value.unwrap_or_default();
        let message_id = msg_offset.to_string();

        let mut headers = HashMap::new();
        for (k, v) in record_and_offset.record.headers {
            if let Ok(val_str) = std::str::from_utf8(&v) {
                headers.insert(k, val_str.to_string());
            }
        }

        let acknowledger = Arc::new(KafkaAcknowledger {
            offset: msg_offset,
            fetch_offset: self.fetch_offset.clone(),
            committed_offset: self.committed_offset.clone(),
            prefetch_buffer: self.prefetch_buffer.clone(),
        });

        ConsumedMessage {
            message_id,
            topic: self.input_topic.clone(),
            payload,
            headers,
            acknowledger,
        }
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
        // Attempt to access and cache partition 0 client; if not present and auto.create.topics is enabled,
        // broker creates it, or controller handles topic provisioning.
        match self.get_or_create_partition_client(topic_name).await {
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
        let buffered = {
            let mut guard = self
                .prefetch_buffer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.pop_front()
        };
        if let Some(record_and_offset) = buffered {
            return Ok(Some(self.build_consumed_message(record_and_offset)));
        }

        let pc = match self.get_or_create_partition_client(&self.input_topic).await {
            Ok(pc) => pc,
            Err(_) => return Ok(None),
        };

        let current_offset = self.fetch_offset.load(Ordering::Acquire);
        let max_wait_ms = i32::try_from(timeout.as_millis()).unwrap_or(200);

        // Fetch records starting at current_offset with timeout
        let fetch_res = pc
            .fetch_records(current_offset, 1..1_048_576, max_wait_ms)
            .await;

        match fetch_res {
            Ok((records, _high_watermark)) => {
                let mut valid_records = records.into_iter().filter(|r| r.offset >= current_offset);

                if let Some(first_record) = valid_records.next() {
                    let msg = self.build_consumed_message(first_record);
                    {
                        let mut guard = self
                            .prefetch_buffer
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        guard.extend(valid_records);
                    }
                    Ok(Some(msg))
                } else {
                    Ok(None)
                }
            }
            Err(e) => {
                // If offset is out of range, seek to earliest
                let err_str = e.to_string();
                if err_str.contains("OffsetOutOfRange") {
                    if let Ok(earliest) = pc.get_offset(OffsetAt::Earliest).await {
                        self.fetch_offset.store(earliest, Ordering::Release);
                    }
                }
                Ok(None)
            }
        }
    }

    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<(), InferenceError> {
        let pc = self.get_or_create_partition_client(topic).await?;

        let record = Record {
            key: None,
            value: Some(payload.to_vec()),
            headers: std::collections::BTreeMap::new(),
            timestamp: rskafka::chrono::Utc::now(),
        };

        pc.produce(vec![record], Compression::NoCompression)
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to produce Kafka message to '{topic}': {e}"
                ))
            })?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn dummy_record(offset: i64, payload: &[u8]) -> RecordAndOffset {
        RecordAndOffset {
            record: Record {
                key: None,
                value: Some(payload.to_vec()),
                headers: BTreeMap::new(),
                timestamp: rskafka::chrono::Utc::now(),
            },
            offset,
        }
    }

    #[tokio::test]
    async fn test_kafka_acknowledger_decouples_fetch_and_committed_offsets() {
        let fetch_offset = Arc::new(AtomicI64::new(3));
        let committed_offset = Arc::new(AtomicI64::new(0));
        let prefetch_buffer = Arc::new(Mutex::new(VecDeque::from(vec![
            dummy_record(1, b"m1"),
            dummy_record(2, b"m2"),
        ])));

        let ack_0 = KafkaAcknowledger {
            offset: 0,
            fetch_offset: fetch_offset.clone(),
            committed_offset: committed_offset.clone(),
            prefetch_buffer: prefetch_buffer.clone(),
        };

        // Positive ack advances committed_offset without rewinding fetch_offset or clearing buffer
        ack_0.ack().await.unwrap();
        assert_eq!(committed_offset.load(Ordering::Acquire), 1);
        assert_eq!(fetch_offset.load(Ordering::Acquire), 3);
        assert_eq!(prefetch_buffer.lock().unwrap().len(), 2);

        // Nack with requeue=true clears buffered records and rewinds fetch_offset to failed offset
        let ack_1 = KafkaAcknowledger {
            offset: 1,
            fetch_offset: fetch_offset.clone(),
            committed_offset: committed_offset.clone(),
            prefetch_buffer: prefetch_buffer.clone(),
        };
        ack_1.nack(true).await.unwrap();
        assert_eq!(fetch_offset.load(Ordering::Acquire), 1);
        assert!(prefetch_buffer.lock().unwrap().is_empty());
        assert_eq!(committed_offset.load(Ordering::Acquire), 1);
    }
}
