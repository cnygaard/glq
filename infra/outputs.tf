output "instance_ip" {
  value       = aws_instance.gpu.public_ip
  description = "Public IP of the GPU instance"
}

output "ssh_command" {
  value       = "ssh -i ${path.module}/gpu-key.pem ubuntu@${aws_instance.gpu.public_ip}"
  description = "SSH command to connect"
}

output "scp_upload" {
  value       = "scp -i ${path.module}/gpu-key.pem -r ../eval_ppl.py ../golay-leech-quant-prototype-v2.py ubuntu@${aws_instance.gpu.public_ip}:~/"
  description = "Copy source files to instance"
}

output "instance_id" {
  value       = aws_instance.gpu.id
  description = "Instance ID (for manual termination)"
}
