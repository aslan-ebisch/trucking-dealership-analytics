
# Organize all resources for IronLedger Analytics
resource "azurerm_resource_group" "ironledger" {
  name     = "rg-${var.project_name}-${var.environment}"
  location = var.location
}

# App Service computing infrastructure
resource "azurerm_service_plan" "ironledger" {
  name                = "asp-${var.project_name}-${var.environment}"
  resource_group_name = azurerm_resource_group.ironledger.name
  location            = var.app_location

  os_type  = "Linux"
  sku_name = "F1"
}

# Linux web application for Streamlit
resource "azurerm_linux_web_app" "ironledger" {
  name                = "${var.project_name}-${var.environment}-web"
  resource_group_name = azurerm_resource_group.ironledger.name
  location            = var.app_location
  service_plan_id     = azurerm_service_plan.ironledger.id

  https_only = true

  site_config {
    always_on          = false
    websockets_enabled = true

    application_stack {
      python_version = "3.11"
    }

    app_command_line = "python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8000 --server.headless true"
  }

  app_settings = {
    SCM_DO_BUILD_DURING_DEPLOYMENT     = "true"
    WEBSITES_PORT                      = "8000"
    STREAMLIT_SERVER_FILE_WATCHER_TYPE = "none"
  }
}



