import { useState, useEffect, useMemo, useRef } from 'react';
import styles from './styles/activationEditor.module.css';

interface ActivationEditorProps {
  code: string;
  normalize: boolean;
  onCodeChange: (params: { code: string; normalize: boolean }) => void;
  presets?: Record<string, string>;
}

export default function ActivationEditor({ code, normalize, onCodeChange, presets = {} }: ActivationEditorProps) {
  const [editedCode, setEditedCode] = useState(code);
  const [normalizeState, setNormalizeState] = useState(normalize);
  const [selectedPreset, setSelectedPreset] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);
  const syntaxHighlightRef = useRef<HTMLDivElement>(null);

  const presetEntries = useMemo(() => {
    const entries = Object.entries(presets);
    const matched = entries.find(([_, presetCode]) => presetCode.trim() === code.trim());
    if (!matched) {
      return [['Currently Active', code], ...entries];
    }
    return entries;
  }, [presets, code]);

  const lineCount = useMemo(() => {
    return editedCode.split('\n').length;
  }, [editedCode]);

  useEffect(() => {
    setEditedCode(code);
  }, [code]);

  useEffect(() => {
    setNormalizeState(normalize);
  }, [normalize]);

  const handleScroll = () => {
    if (textareaRef.current && lineNumbersRef.current && syntaxHighlightRef.current) {
      const scrollTop = textareaRef.current.scrollTop;
      const scrollLeft = textareaRef.current.scrollLeft;
      
      lineNumbersRef.current.scrollTop = scrollTop;
      syntaxHighlightRef.current.scrollTop = scrollTop;
      syntaxHighlightRef.current.scrollLeft = scrollLeft;
    }
  };

  // Handle tab key for proper indentation
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newValue = editedCode.substring(0, start) + '  ' + editedCode.substring(end);
      setEditedCode(newValue);
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 2;
      }, 0);
    }
  };

  // Simple WGSL syntax highlighting with token-based parsing
  const highlightSyntax = (code: string) => {
    function escapeHTML(str: string): string {
      return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    const keywords = new Set(['fn', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return', 'struct', 'uniform', 'vertex', 'fragment', 'compute']);
    const types = new Set(['f32', 'i32', 'u32', 'vec2', 'vec3', 'vec4', 'mat2x2', 'mat3x3', 'mat4x4', 'bool', 'array']);
    const tokens = code.split(/(\s+|\/\/.*$|"(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fh]?|[a-zA-Z_][a-zA-Z0-9_]*|[^\w\s])/gm).filter(token => token !== '');
    
    let result = '';

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      const nextToken = tokens[i + 1];
      
      if (!token) continue;
      
      if (/^\s+$/.test(token)) {
        result += token;
        continue;
      }
      
      if (token.startsWith('//')) {
        result += `<span class="comment">${escapeHTML(token)}</span>`;
        continue;
      }
      
      if (token.startsWith('"')) {
        result += `<span class="string">${token}</span>`;
        continue;
      }
      
      if (/^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fh]?$/.test(token)) {
        result += `<span class="number">${token}</span>`;
        continue;
      }
      
      if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(token) && nextToken === '(') {
        result += `<span class="function">${token}</span>`;
        continue;
      }
      
      if (keywords.has(token)) {
        result += `<span class="keyword">${token}</span>`;
        continue;
      }
      
      if (types.has(token)) {
        result += `<span class="type">${token}</span>`;
        continue;
      }
      
      if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(token)) {
        result += `<span class="other">${token}</span>`;
      } else {
        result += `<span class="ops">${token}</span>`;
      }
    }
    
    return result;
  };

  const onApply = () => {
    onCodeChange({
      code: editedCode,
      normalize: normalizeState,
    });
  };

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = e.target.value;
    setSelectedPreset(selected);
    const presetCode = presetEntries.find(([name]) => name === selected)?.[1];
    if (presetCode !== undefined) {
      setEditedCode(presetCode);
    }
    setSelectedPreset('');
  };

  return (
    <div className={styles.container}>
      <label className={styles.label} htmlFor="presetSelector">
        Load Preset:
      </label>
      <select
        id="presetSelector"
        className={styles.select}
        value={selectedPreset}
        onChange={handlePresetChange}
      >
        <option value="" disabled>Select preset...</option>
        {presetEntries.map(([name]) => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>

      <label className={styles.label} htmlFor="activationCode">
        Activation Function (WGSL):
      </label>
      
      <div className={styles.codeEditor}>
        <div className={styles.lineNumbers} ref={lineNumbersRef}>
          {Array.from({ length: lineCount }, (_, i) => (
            <div key={i + 1} className={styles.lineNumber}>
              {i + 1}
            </div>
          ))}
        </div>
        
        <div className={styles.syntaxHighlight} ref={syntaxHighlightRef}>
          <pre 
            dangerouslySetInnerHTML={{ 
              __html: highlightSyntax(editedCode) + '\n' 
            }}
          />
        </div>
        <textarea
          id="activationCode"
          ref={textareaRef}
          className={styles.codeTextarea}
          value={editedCode}
          onChange={e => setEditedCode(e.target.value)}
          onScroll={handleScroll}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          placeholder="// Enter your WGSL activation function here..."

        />
      </div>
      
      <div className={styles.buttonRow}>
        <button className={styles.button} onClick={onApply}>
          Apply Changes
        </button>
        <label className={styles.checkboxLabel}>
          <input
            type="checkbox"
            className={styles.checkboxInput}
            checked={normalizeState}
            onChange={e => setNormalizeState(e.target.checked)}
          />
          Pre-normalize input by sum of weights
        </label>
      </div>
    </div>
  );
}