import { useState } from 'react';
import styles from './styles/contentSwitcher.module.css';

interface ContentSwitcherProps {
  labels: string[];
  setSelectedIndex?: (number) => void;
  children: React.ReactNode[];
}

export default function ContentSwitcher({ labels, setSelectedIndex, children }: ContentSwitcherProps) {
  const [activeIndex, setActiveIndex] = useState(0);

  const updateActiveIndex = (index) => {
    setActiveIndex(index);
    setSelectedIndex ? setSelectedIndex(index) : null;
  };

  return (
    <div className={styles.switcherContainer}>
      <div className={styles.contentBox}>
        <div className={styles.buttonRow}>
          {labels.map((label, idx) => (
            <button
              key={idx}
              className={`${styles.switchButton} ${idx === activeIndex ? styles.switchButtonActive : ''}`}
              onClick={() => updateActiveIndex(idx)}
            >
              {label}
            </button>
          ))}
        </div>
        <div>
          {children[activeIndex]}
        </div>
      </div>
    </div>
  );
}