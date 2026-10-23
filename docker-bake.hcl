variable "IMAGE_NAME" {
  default = "snooze-ml"
}

variable "TAG" {
  default = "latest"
}

variable "VERSION" {
  default = "0.1.1"
}

group "default" {
  targets = ["cuda"]
}

group "all" {
  targets = ["slim", "cpu", "cuda"]
}

target "base" {
  dockerfile = "Dockerfile"
  platforms = [
    "linux/amd64",
    "linux/arm64"
  ]
}

target "slim" {
  inherits = ["base"]
  target = "slim"
  tags = [
    "${IMAGE_NAME}:slim",
    "${IMAGE_NAME}:lite",
    "${IMAGE_NAME}:${VERSION}-slim",
    "${IMAGE_NAME}:${VERSION}-lite"
  ]
}

target "cpu" {
  inherits = ["base"]
  target = "cpu"
  tags = [
    "${IMAGE_NAME}:cpu",
    "${IMAGE_NAME}:${VERSION}-cpu"
  ]
}

target "cuda" {
  inherits = ["base"]
  target = "cuda"
  platforms = [
    "linux/amd64"
  ]
  tags = [
    "${IMAGE_NAME}:${TAG}",
    "${IMAGE_NAME}:${VERSION}",
    "${IMAGE_NAME}:cuda",
    "${IMAGE_NAME}:full",
    "${IMAGE_NAME}:${VERSION}-cuda",
    "${IMAGE_NAME}:${VERSION}-full"
  ]
}
