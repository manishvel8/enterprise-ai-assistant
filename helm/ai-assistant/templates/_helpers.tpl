{{/*
_helpers.tpl — Reusable template helpers.

These are named templates (like functions) that can be called from any template.
Define once here, call everywhere → DRY principle in Helm.

Usage in a template:
  {{ include "ai-assistant.labels" . }}
  {{ include "ai-assistant.selectorLabels" "backend" }}
*/}}

{{/*
Common labels applied to all resources.
These help kubectl filter: kubectl get all -l chart=ai-assistant
*/}}
{{- define "ai-assistant.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
release: {{ .Release.Name }}
{{- end }}

{{/*
Selector labels (used in matchLabels and Service selector).
Must be stable — changing these requires a new deployment.
*/}}
{{- define "ai-assistant.selectorLabels" -}}
app: {{ . }}
{{- end }}

{{/*
Image name helper with optional registry prefix.
Usage: {{ include "ai-assistant.image" (dict "image" .Values.backend.image "tag" .Values.global.imageTag) }}
*/}}
{{- define "ai-assistant.image" -}}
{{- if .registry -}}
{{ .registry }}/{{ .image }}:{{ .tag }}
{{- else -}}
{{ .image }}:{{ .tag }}
{{- end -}}
{{- end }}

{{/*
Namespace helper.
*/}}
{{- define "ai-assistant.namespace" -}}
{{ .Values.global.namespace | default "ai-assistant" }}
{{- end }}
