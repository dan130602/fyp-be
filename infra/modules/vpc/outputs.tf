output "vpc_id" {
    value = aws_vpc.this.id
}

output "public_subnet_ids" {
    value = {
        for k, v in aws_subnet.public : k => v.id
    }
}

output "private_subnet_ids" {
    value = {
        for k, v in aws_subnet.private : k => v.id
    }
}

output "public_route_table_id" {
    value = length(aws_route_table.public) > 0 ? aws_route_table.public[0].id : null
}

output "private_route_table_ids" {
    value = {
        for k, v in aws_route_table.private : k => v.id
    }
}

output "internet_gateway_id" {
    value = length(aws_internet_gateway.this) > 0 ? aws_internet_gateway.this[0].id : null
}

output "s3_vpc_endpoint_id" {
    value = var.create_s3_gateway_endpoint ? aws_vpc_endpoint.s3[0].id : null
}