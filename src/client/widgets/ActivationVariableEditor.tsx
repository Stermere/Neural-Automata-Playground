import { useEffect, useState } from 'react';
import styles from './styles/activationVariableEditor.module.css';
import { ActivationVariableController, ActivationVariable, VariableValue } from '../controllers/ActivationVariableController';

interface ActivationVariableEditorProps {
  code: string;
  onVariableChange: (code: string) => void;
}

export default function ActivationVariableEditor({ code, onVariableChange }: ActivationVariableEditorProps) {
  const [variables, setVariables] = useState<ActivationVariable[]>([]);
  const [values, setValues] = useState<VariableValue[]>([]);
  const [debouncedValues, setDebouncedValues] = useState<VariableValue[]>([]);

  useEffect(() => {
    const vars = ActivationVariableController.getVariables(code);
    const initialValues = vars.map((v) => ({
      name: v.name,
      value: v.default.toString(),
    }));
    setVariables(vars);
    setValues(initialValues);
    setDebouncedValues(initialValues);
  }, [code]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setDebouncedValues(values);
    }, 100);
    return () => clearTimeout(timeout);
  }, [values]);

  useEffect(() => {
    const newCode = ActivationVariableController.transformActivationCode(code, debouncedValues);
    onVariableChange(newCode);
  }, [debouncedValues]);

  const handleSliderChange = (name: string, newValue: string) => {
    setValues((prev) =>
      prev.map((v) => (v.name === name ? { ...v, value: newValue } : v))
    );
  };

  return (
    <div className={styles.container}>
      {variables.map((v) => {
        const currentValue = values.find((val) => val.name === v.name)?.value ?? v.default.toString();
        return (
          <div key={v.name} className={styles.output}>
            <label className={styles.label} htmlFor={`slider-${v.name}`}>
              {v.name}: {parseFloat(currentValue).toFixed(2)}
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