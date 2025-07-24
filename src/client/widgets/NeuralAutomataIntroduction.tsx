import ReactMarkdown from 'react-markdown';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import oneDark from 'react-syntax-highlighter/dist/esm/styles/prism/one-dark';

import styles from './styles/neuralAutomataIntroduction.module.css';
import neuralAutomataIntro from '../assets/neuralAutomataIntro.md?raw';
import convolutionDiagram from '/convolutionDiagram.png?url';

export default function NeuralAutomataIntroduction() {
  return (
    <div className={styles.container}>
      <ReactMarkdown
        components={{
          code({ children, ...props }) {
            return (
              <SyntaxHighlighter
                language="rust"
                style={oneDark}
                PreTag="div"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            );
          },
        }}
      >
        {neuralAutomataIntro}
      </ReactMarkdown>
      <img className={styles.math} src={convolutionDiagram} loading="lazy"/>
    </div>
  );
}