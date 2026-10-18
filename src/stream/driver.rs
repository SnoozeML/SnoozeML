//! Pluggable driver interface for asynchronous message stream brokers.

use crate::types::InferenceError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Trait representing an acknowledgment callback for a consumed stream message.
#[async_trait]
pub trait MessageAcknowledger: Send + Sync {
    /// Positively acknowledges that the message was processed and published successfully.
    ///
    /// # Errors
    /// Returns [`InferenceError`] if acknowledgment fails on the broker.
    async fn ack(&self) -> Result<(), InferenceError>;

    /// Negatively acknowledges the message, optionally requesting broker redelivery.
    ///
    /// # Errors
    /// Returns [`InferenceError`] if nack fails on the broker.
    async fn nack(&self, requeue: bool) -> Result<(), InferenceError>;
}

/// A consumed stream message containing raw payload bytes, metadata, and an acknowledgment handle.
pub struct ConsumedMessage {
    /// Unique broker-specific message identifier.
    pub message_id: String,
    /// Target topic, stream key, or queue name from which this message was consumed.
    pub topic: String,
    /// Raw unparsed payload bytes.
    pub payload: Vec<u8>,
    /// Optional message headers or broker properties.
    pub headers: HashMap<String, String>,
    /// Acknowledgment handle for marking message processed or rejected.
    pub acknowledger: Arc<dyn MessageAcknowledger>,
}

impl std::fmt::Debug for ConsumedMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsumedMessage")
            .field("message_id", &self.message_id)
            .field("topic", &self.topic)
            .field("payload_len", &self.payload.len())
            .field("headers", &self.headers)
            .finish()
    }
}

/// Pluggable asynchronous driver interface implemented by Kafka, RabbitMQ, Redis, and mock streams.
#[async_trait]
pub trait StreamDriver: Send + Sync {
    /// Returns the human-readable broker identifier (e.g. `"kafka"`, `"rabbitmq"`, `"redis"`, `"mock"`).
    fn broker_name(&self) -> &'static str;

    /// Automatically ensures that input, output, or error topics/queues exist on the broker.
    ///
    /// If the topic or queue does not exist and automatic creation is permitted,
    /// this function provisions it. If it already exists, this is a no-op.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] or [`InferenceError::InvalidInput`] if provisioning fails.
    async fn ensure_topic_exists(
        &self,
        topic_name: &str,
        partitions: usize,
    ) -> Result<(), InferenceError>;

    /// Polls for the next incoming message from the subscribed input topic/queue.
    ///
    /// This function is non-blocking with respect to the Tokio reactor: it awaits asynchronously
    /// for up to `timeout` and returns `Ok(None)` if no message is available within the window.
    ///
    /// # Errors
    /// Returns [`InferenceError`] if a transport or protocol failure occurs.
    async fn poll_next(&self, timeout: Duration)
        -> Result<Option<ConsumedMessage>, InferenceError>;

    /// Publishes an inference response payload to the destination topic/queue.
    ///
    /// # Errors
    /// Returns [`InferenceError`] if the broker rejects the message or connection drops.
    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<(), InferenceError>;
}
