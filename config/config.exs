import Config

config :nx, :default_backend, EXLA.Backend

#config :exla, :clients,
#  host: [platform: :host]

config :exla,
  clients: [
    host: [platform: :host]
  ],
  preferred_clients: [:host]

import_config "#{config_env()}.exs"
