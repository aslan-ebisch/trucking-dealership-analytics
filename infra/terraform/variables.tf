
variable "location" {
  description = "Azure region for IronLedger resources"
  type        = string
  default     = "centralus"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Name of the application"
  type        = string
  default     = "ironledger"
}

variable "subscription_id" {
  description = "Azure subscription used for deployment"
  type        = string
  default     = "943690a2-1f14-4e90-904c-51ed81e219b8"
}
