/** Tiled server configuration returned by `GET /api/config/servers`. */
export interface ServerInfo {
  name: string;
  uri: string;
  has_api_key: boolean;
  api_key?: string;
}
