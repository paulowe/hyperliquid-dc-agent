resource "google_artifact_registry_repository" "kfp_base" {
  project       = var.project_id
  repository_id = var.repo_id
  description   = "DC VAE docker repository for some KFP components"
  location      = var.region
  format        = "DOCKER"
}
