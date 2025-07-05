import { DEFAULT_CONFIG } from "../constants/filenameConstants";

export type ConfigData = {
  weights: number[][][][];
  activationCode: string;
  normalize: boolean;
};

const configModules = import.meta.glob('../constants/defaultConfigs/*.json', { eager: true });

export class DefaultConfigController {
  static configMap: Record<string, ConfigData> = Object.fromEntries(
    Object.entries(configModules).map(([path, mod]) => {
      const name = path.match(/\/([^/]+)\.json$/)?.[1];
      return [name!, (mod as any).default as ConfigData];
    })
  );

  static getAvailableConfigNames(): string[] {
    return Object.keys(this.configMap);
  }

  static getConfig(name: string): ConfigData | undefined {
    return this.configMap[name];
  }

  static getDefault(): ConfigData {
    return this.getConfig(DEFAULT_CONFIG) ?? Object.values(this.configMap)[0];
  }
}