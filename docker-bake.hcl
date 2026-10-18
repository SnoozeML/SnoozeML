variable "TAG" {
  default = "latest"
}

variable "VERSION" {
  default = "0.1.0"
}

group "default" {
  targets = ["cpu"]
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
    "snooze-ml:slim",
    "snooze-ml:${VERSION}-slim"
  ]
}

target "cpu" {
  inherits = ["base"]
  target = "cpu"
  tags = [
    "snooze-ml:cpu",
    "snooze-ml:${TAG}",
    "snooze-ml:${VERSION}"
  ]
}

target "cuda" {
  inherits = ["base"]
  target = "cuda"
  platforms = [
    "linux/amd64"
  ]
  tags = [
    "snooze-ml:cuda",
    "snooze-ml:${TAG}-cuda",
    "snooze-ml:${VERSION}-cuda"
  ]
}
