variable "region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1" # Stockholm
}

variable "instance_type" {
  description = "GPU instance type"
  type        = string
  default     = "g5.xlarge" # A10G 24GB, 4 vCPU, 16GB RAM
}
