/** Tiled server configuration returned by `GET /api/config/servers`.
 *
 * The backend never returns the credential itself (see `ServerConfig` in
 * annotation_server.py) — only whether one is configured.
 */
export interface ServerInfo {
  name: string;
  uri: string;
  has_api_key: boolean;
}
