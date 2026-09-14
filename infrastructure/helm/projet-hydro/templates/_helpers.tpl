{{- define "projet-hydro.fullname" -}}
{{ .Chart.Name }}
{{- end -}}

{{- define "projet-hydro.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
