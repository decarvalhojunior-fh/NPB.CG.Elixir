import Config

config :nx, :default_backend, {EXLA.Backend, client: :host}

config :exla, :clients,
  host: [platform: :host]
