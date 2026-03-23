import Config

config :nx, :default_backend, EXLA.Backend

config :exla, :clients,
  host: [platform: :host]

import_config "#{config_env()}.exs"
