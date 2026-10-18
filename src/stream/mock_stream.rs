//! In-memory channel-based stream driver for testing and deterministic verification.

use crate::stream::driver::{ConsumedMessage, MessageAcknowledger, StreamDriver};
use crate::types::InferenceError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex, RwLock};

struct MockAcknowledger {
    ack_counter: Arc<AtomicU64>,
    nack_counter: Arc<AtomicU64>,
}

#[async_trait]
impl MessageAcknowledger for MockAcknowledger {
    async fn ack(&self) -> Result<(), InferenceError> {
        self.ack_counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn nack(&self, _requeue: bool) -> Result<(), InferenceError> {
        self.nack_counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// In-memory mock stream driver backed by Tokio asynchronous channels.
#[derive(Clone)]
pub struct MockStreamDriver {
    input_topic: String,
    topics: Arc<RwLock<HashMap<String, mpsc::Sender<Vec<u8>>>>>,
    receivers: Arc<Mutex<HashMap<String, mpsc::Receiver<Vec<u8>>>>>,
    published: Arc<RwLock<HashMap<String, Vec<Vec<u8>>>>>,
    ack_count: Arc<AtomicU64>,
    nack_count: Arc<AtomicU64>,
    msg_counter: Arc<AtomicU64>,
}

impl MockStreamDriver {
    /// Creates a new `MockStreamDriver` listening to `input_topic`.
    #[must_use]
    pub fn new(input_topic: impl Into<String>) -> Self {
        Self {
            input_topic: input_topic.into(),
            topics: Arc::new(RwLock::new(HashMap::new())),
            receivers: Arc::new(Mutex::new(HashMap::new())),
            published: Arc::new(RwLock::new(HashMap::new())),
            ack_count: Arc::new(AtomicU64::new(0)),
            nack_count: Arc::new(AtomicU64::new(0)),
            msg_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Injects a message into a topic (simulating an external producer).
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if the topic channel is closed or does not exist.
    pub async fn inject_message(
        &self,
        topic: &str,
        payload: Vec<u8>,
    ) -> Result<(), InferenceError> {
        let guard = self.topics.read().await;
        if let Some(tx) = guard.get(topic) {
            tx.send(payload).await.map_err(|e| {
                InferenceError::Runtime(format!("Failed to inject into mock topic '{topic}': {e}"))
            })?;
            Ok(())
        } else {
            Err(InferenceError::NotFound(format!(
                "Mock topic '{topic}' does not exist"
            )))
        }
    }

    /// Retrieves all published messages for a given topic.
    pub async fn get_published(&self, topic: &str) -> Vec<Vec<u8>> {
        let guard = self.published.read().await;
        guard.get(topic).cloned().unwrap_or_default()
    }

    /// Returns the total count of successfully acknowledged messages.
    #[must_use]
    pub fn ack_count(&self) -> u64 {
        self.ack_count.load(Ordering::Relaxed)
    }

    /// Returns the total count of negatively acknowledged (rejected) messages.
    #[must_use]
    pub fn nack_count(&self) -> u64 {
        self.nack_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl StreamDriver for MockStreamDriver {
    fn broker_name(&self) -> &'static str {
        "mock"
    }

    async fn ensure_topic_exists(
        &self,
        topic_name: &str,
        _partitions: usize,
    ) -> Result<(), InferenceError> {
        let mut topics = self.topics.write().await;
        if !topics.contains_key(topic_name) {
            let (tx, rx) = mpsc::channel(1024);
            topics.insert(topic_name.to_string(), tx);
            let mut receivers = self.receivers.lock().await;
            receivers.insert(topic_name.to_string(), rx);
        }
        Ok(())
    }

    async fn poll_next(
        &self,
        timeout: Duration,
    ) -> Result<Option<ConsumedMessage>, InferenceError> {
        let mut receivers = self.receivers.lock().await;
        let rx = match receivers.get_mut(&self.input_topic) {
            Some(r) => r,
            None => return Ok(None),
        };

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(payload)) => {
                let id = self.msg_counter.fetch_add(1, Ordering::Relaxed);
                let message_id = format!("mock-msg-{id}");
                let acknowledger = Arc::new(MockAcknowledger {
                    ack_counter: self.ack_count.clone(),
                    nack_counter: self.nack_count.clone(),
                });

                Ok(Some(ConsumedMessage {
                    message_id,
                    topic: self.input_topic.clone(),
                    payload,
                    headers: HashMap::new(),
                    acknowledger,
                }))
            }
            Ok(None) => Ok(None),
            Err(_) => Ok(None), // Polling timeout elapsed
        }
    }

    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<(), InferenceError> {
        let mut guard = self.published.write().await;
        guard
            .entry(topic.to_string())
            .or_default()
            .push(payload.to_vec());
        Ok(())
    }
}
