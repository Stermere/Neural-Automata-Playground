import { useEffect, useState } from 'react';
import { useDebounce } from '../hooks/useDebounce';
import styles from './styles/activationVariableEditor.module.css';
import { ActivationVariableUtils, ActivationVariable, VariableValue } from '../utils/ActivationVariableUtils';

interface ActivationVariableEditorProps {
  code: string;
  values: VariableValue[],
  setValues: (values: VariableValue[]) => void;
  onVariableChange: (code: string) => void;
}

export default function ActivationVariableEditor({ code, values, setValues, onVariableChange }: ActivationVariableEditorProps) {
  const [variables, setVariables] = useState<ActivationVariable[]>([]);

  const debouncedValues = useDebounce(values, 100);

  useEffect(() => {
    const parsedVars = ActivationVariableUtils.getVariables(code);

    // Build new values based on memory or default
    const newValues: VariableValue[] = parsedVars.map((v) => {
      const remembered = values.find((val) => val.name === v.name);
      return {
        name: v.name,
        value: remembered?.value ?? v.default,
      };
    });

    // Save to state and memory
    setVariables(parsedVars);
    setValues(newValues);
  }, [code]);


  useEffect(() => {
    const newCode = ActivationVariableUtils.transformActivationCode(code, values);
    onVariableChange(newCode);
  }, [debouncedValues]);

  const handleSliderChange = (name: string, newValue: string) => {
    setValues(values.map((v) => (v.name === name ? { ...v, value: parseFloat(newValue) } : v))
    );
  };

  return (
    <div className={styles.container}>
      {variables.map((v) => {
        const currentValue = values.find((val) => val.name === v.name)?.value ?? v.default;
        return (
          <div key={v.name} className={styles.output}>
            <label className={styles.label} htmlFor={`slider-${v.name}`}>
              {v.name}: {currentValue.toFixed(2)}
            </label>
            <input
              id={`slider-${v.name}`}
              type="range"
              min={v.min}
              max={v.max}
              step="0.01"
              value={currentValue}
              onChange={(e) => handleSliderChange(v.name, e.target.value)}
              className={styles.slider}
            />
          </div>
        );
      })}
    </div>
  );
}