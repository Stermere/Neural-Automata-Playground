export type ShareableConfig = {
  weights: number[][][][];
  activationCode: string;
  normalize: boolean;
  computeKernel?: boolean;
};

const HASH_PARAM = 'cfg';

// Encodes a full config into the URL hash (deflate + base64url) so patterns
// can be shared as plain links. The hash never reaches the server, so this
// works on static hosting like GitHub Pages.
export class ShareUtils {
  static async buildShareUrl(config: ShareableConfig): Promise<string> {
    const encoded = await this.encode(config);
    const url = new URL(window.location.href);
    url.hash = `${HASH_PARAM}=${encoded}`;
    return url.toString();
  }

  static async loadFromUrl(): Promise<ShareableConfig | null> {
    const prefix = `#${HASH_PARAM}=`;
    const hash = window.location.hash;
    if (!hash.startsWith(prefix)) return null;

    try {
      const config = await this.decode(hash.slice(prefix.length));
      if (!Array.isArray(config.weights) || typeof config.activationCode !== 'string') {
        throw new Error('Config is missing weights or activation code');
      }
      return config;
    } catch (err) {
      console.error('Failed to load shared config from URL:', err);
      return null;
    }
  }

  private static async encode(config: ShareableConfig): Promise<string> {
    const bytes = new TextEncoder().encode(JSON.stringify(config));
    const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
    const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
    return this.toBase64Url(compressed);
  }

  private static async decode(encoded: string): Promise<ShareableConfig> {
    const compressed = this.fromBase64Url(encoded);
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    const json = await new Response(stream).text();
    return JSON.parse(json);
  }

  private static toBase64Url(bytes: Uint8Array): string {
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  private static fromBase64Url(encoded: string): Uint8Array {
    const base64 = encoded.replace(/-/g, '+').replace(/_/g, '/');
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }
}
