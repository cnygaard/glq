terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- SSH key ---

resource "tls_private_key" "ssh" {
  algorithm = "ED25519"
}

resource "aws_key_pair" "gpu" {
  key_name   = "golay-leech-quant-${random_id.suffix.hex}"
  public_key = tls_private_key.ssh.public_key_openssh
}

resource "local_file" "ssh_key" {
  content         = tls_private_key.ssh.private_key_openssh
  filename        = "${path.module}/gpu-key.pem"
  file_permission = "0600"
}

resource "random_id" "suffix" {
  byte_length = 4
}

# --- Security group ---

resource "aws_security_group" "gpu" {
  name_prefix = "golay-leech-quant-"
  description = "SSH access for GPU eval instance"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "golay-leech-quant" }
}

# --- AMI: Deep Learning Base (Ubuntu 22.04, NVIDIA drivers pre-installed) ---

data "aws_ami" "dl" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*"]
    #values = ["Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) ????????"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# --- On-demand instance: g5.xlarge (A10G 24GB) ---

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.dl.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.gpu.key_name
  vpc_security_group_ids = [aws_security_group.gpu.id]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/setup.sh.tftpl", {
    repo_tar_b64 = "" # We'll scp the code instead
  })

  tags = { Name = "golay-leech-quant-eval" }
}
