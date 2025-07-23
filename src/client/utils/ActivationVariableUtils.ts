export type ActivationVariable = {
  name: string;
  default: number;
  max: number;
  min: number;
};

export type VariableValue = {
  name: string;
  value: number;
};

const CONST_F32_MATCHER: RegExp = /const\s+(\w+)\s*:\s*f32\s*=\s*([\d.eE+-]+)\s*;\s*@variable\s+([\d.eE+-]+)\s+([\d.eE+-]+)/g;

export class ActivationVariableUtils {
  // Finds lines with constants and @variable annotations, replaces with defaults
  static transformActivationCodeDefault(code: string): string {
    return code.replace(
      CONST_F32_MATCHER,
      (match, name, def, max, min) => `const ${name}: f32 = ${def};`
    );
  }

  // Replaces variable values with user-modified values
  static transformActivationCode(code: string, variableValues: VariableValue[]): string {
    return code.replace(
      CONST_F32_MATCHER,
      (match, name, def, max, min) => {
        const variable = variableValues.find((v) => v.name === name);
        const newValue = variable ? variable.value : def;
        return `const ${name}: f32 = ${newValue};`;
      }
    );
  }

  // Extracts all @variable constants into ActivationVariable objects
  static getVariables(code: string): ActivationVariable[] {
    const variables: ActivationVariable[] = [];
    let match;
    while ((match = CONST_F32_MATCHER.exec(code)) !== null) {
      const [_, name, defStr, minStr, maxStr] = match;
      variables.push({
        name,
        default: parseFloat(defStr),
        max: parseFloat(maxStr),
        min: parseFloat(minStr),
      });
    }

    return variables;
  }

  static getDefaultVariableValues(activationVariables: ActivationVariable[]): VariableValue[] {
    return activationVariables.map((variable) => ({
      name: variable.name,
      value: variable.default,
    }));
  }

  static updateDefaults(code: string, variableValues: VariableValue[]): string {
    return code.replace(
      CONST_F32_MATCHER,
      (_match, name, defStr, minStr, maxStr) => {
        const variable = variableValues.find((v) => v.name === name);
        const newDefault = variable != null ? variable.value : parseFloat(defStr);

        return `const ${name}: f32 = ${newDefault}; @variable ${minStr} ${maxStr}`;
      }
    );
  }
}