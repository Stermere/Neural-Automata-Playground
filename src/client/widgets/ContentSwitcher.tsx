import { useState } from 'react';
import styles from './styles/contentSwitcher.module.css';

interface ContentSwitcherProps {
  labels: string[];
  children: React.ReactNode[];
}

export default function ContentSwitcher({ labels, children }: ContentSwitcherProps) {
  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <div className={styles.switcherContainer}>
      <div className={styles.contentBox}>
        <div className={styles.buttonRow}>
          {labels.map((label, idx) => (
            <button
              key={idx}
              className={`${styles.switchButton} ${idx === activeIndex ? styles.switchButtonActive : ''}`}
              onClick={() => setActiveIndex(idx)}
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