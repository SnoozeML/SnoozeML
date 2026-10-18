//! RabbitMQ AMQP 0-9-1 stream driver implementation using `lapin`.

use crate::stream::driver::{ConsumedMessage, MessageAcknowledger, StreamDriver};
use crate::types::InferenceError;
use async_trait::async_trait;
use lapin::options::{
    BasicAckOptions, BasicConsumeOptions, BasicNackOptions, BasicPublishOptions, BasicQosOptions,
    QueueDeclareOptions,
};
use lapin::types::FieldTable;
use lapin::{BasicProperties, Channel, Connection, ConnectionProperties, Consumer};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

struct LapinAcknowledger {
    delivery_tag: u64,
    channel: Channel,
}

#[async_trait]
impl MessageAcknowledger for LapinAcknowledger {
    async fn ack(&self) -> Result<(), InferenceError> {
        self.channel
            .basic_ack(self.delivery_tag, BasicAckOptions::default())
            .await
            .map_err(|e| InferenceError::Runtime(format!("RabbitMQ basic_ack failed: {e}")))
    }

    async fn nack(&self, requeue: bool) -> Result<(), InferenceError> {
        self.channel
            .basic_nack(
                self.delivery_tag,
                BasicNackOptions {
                    multiple: false,
                    requeue,
                },
            )
            .await
            .map_err(|e| InferenceError::Runtime(format!("RabbitMQ basic_nack failed: {e}")))
    }
}

/// RabbitMQ AMQP 0-9-1 stream driver.
pub struct RabbitMqDriver {
    input_queue: String,
    channel: Channel,
    consumer: Arc<Mutex<Option<Consumer>>>,
}

impl RabbitMqDriver {
    /// Connects to a RabbitMQ AMQP server and initializes the channel and QoS.
    ///
    /// # Errors
    /// Returns [`InferenceError::Runtime`] if connecting to AMQP fails.
    pub async fn connect(
        endpoint: &str,
        input_queue: impl Into<String>,
        prefetch_count: u16,
    ) -> Result<Self, InferenceError> {
        let conn = Connection::connect(endpoint, ConnectionProperties::default())
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to connect to RabbitMQ at '{endpoint}': {e}"
                ))
            })?;

        let channel = conn.create_channel().await.map_err(|e| {
            InferenceError::Runtime(format!("Failed to open RabbitMQ channel: {e}"))
        })?;

        channel
            .basic_qos(prefetch_count, BasicQosOptions::default())
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!("Failed to set RabbitMQ basic_qos: {e}"))
            })?;

        Ok(Self {
            input_queue: input_queue.into(),
            channel,
            consumer: Arc::new(Mutex::new(None)),
        })
    }

    async fn get_or_create_consumer(&self) -> Result<Consumer, InferenceError> {
        let mut guard = self.consumer.lock().await;
        if let Some(ref c) = *guard {
            return Ok(c.clone());
        }

        let consumer = self
            .channel
            .basic_consume(
                &self.input_queue,
                "snoozeml_consumer",
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to start RabbitMQ consumer on '{}': {e}",
                    self.input_queue
                ))
            })?;

        *guard = Some(consumer.clone());
        Ok(consumer)
    }
}

#[async_trait]
impl StreamDriver for RabbitMqDriver {
    fn broker_name(&self) -> &'static str {
        "rabbitmq"
    }

    async fn ensure_topic_exists(
        &self,
        topic_name: &str,
        _partitions: usize,
    ) -> Result<(), InferenceError> {
        // Declaring a queue with passive: false automatically creates it if it does not exist
        let options = QueueDeclareOptions {
            passive: false,
            durable: true,
            exclusive: false,
            auto_delete: false,
            nowait: false,
        };

        self.channel
            .queue_declare(topic_name, options, FieldTable::default())
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to declare RabbitMQ queue '{topic_name}': {e}"
                ))
            })?;

        Ok(())
    }

    async fn poll_next(
        &self,
        timeout: Duration,
    ) -> Result<Option<ConsumedMessage>, InferenceError> {
        let mut consumer = self.get_or_create_consumer().await?;

        // Read next delivery from consumer stream with timeout
        use futures_lite::StreamExt;
        let delivery_res = tokio::time::timeout(timeout, consumer.next()).await;

        match delivery_res {
            Ok(Some(Ok(delivery))) => {
                let message_id = delivery.delivery_tag.to_string();
                let payload = delivery.data;

                let mut headers = HashMap::new();
                if let Some(corr_id) = delivery.properties.correlation_id() {
                    headers.insert("correlation_id".to_string(), corr_id.to_string());
                }
                if let Some(reply_to) = delivery.properties.reply_to() {
                    headers.insert("reply_to".to_string(), reply_to.to_string());
                }

                let acknowledger = Arc::new(LapinAcknowledger {
                    delivery_tag: delivery.delivery_tag,
                    channel: self.channel.clone(),
                });

                Ok(Some(ConsumedMessage {
                    message_id,
                    topic: self.input_queue.clone(),
                    payload,
                    headers,
                    acknowledger,
                }))
            }
            Ok(Some(Err(e))) => Err(InferenceError::Runtime(format!(
                "RabbitMQ delivery stream error: {e}"
            ))),
            Ok(None) => Ok(None),
            Err(_) => Ok(None), // Polling timeout elapsed
        }
    }

    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<(), InferenceError> {
        self.channel
            .basic_publish(
                "",
                topic,
                BasicPublishOptions::default(),
                payload,
                BasicProperties::default(),
            )
            .await
            .map_err(|e| {
                InferenceError::Runtime(format!(
                    "Failed to publish RabbitMQ message to '{topic}': {e}"
                ))
            })?;

        Ok(())
    }
}
