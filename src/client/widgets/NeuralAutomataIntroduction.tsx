import ReactMarkdown from 'react-markdown';
import styles from './styles/neuralAutomataIntroduction.module.css';
import neuralAutomataIntro from '../assets/neuralAutomataIntro.md?raw';
import convolutionDiagram from '../assets/convolutionDiagram.png';

export default function NeuralAutomataIntroduction() {
  return (
    <div className={styles.container}>
      <ReactMarkdown>{neuralAutomataIntro}</ReactMarkdown>
      <img className={styles.math} src={convolutionDiagram} />
    </div>
  );
}