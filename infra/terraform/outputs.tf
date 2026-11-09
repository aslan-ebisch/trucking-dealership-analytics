
output "resource_group_name" {
  description = "Name of the deployed resource group"
  value       = azurerm_resource_group.ironledger.name
}

output "web_app_name" {
  description = "Name of the IronLedger web application"
  value       = azurerm_linux_web_app.ironledger.name
}

output "web_app_url" {
  description = "Public URL of the deployed application"
  value       = "https://${azurerm_linux_web_app.ironledger.default_hostname}"
}
