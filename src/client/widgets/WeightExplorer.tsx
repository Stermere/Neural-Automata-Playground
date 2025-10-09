import { useEffect, useState } from 'react';
import styles from './styles/weightExplorer.module.css';
import { WeightExplorerController } from '../controllers/WeightExplorerController';
import { VariableValue } from '../utils/ActivationVariableUtils';


interface WeightExplorerProps {
  controller: WeightExplorerController | null;
  updateWeights: (updatedWeights: number[][][][]) => void;
}

export default function WeightExplorer({ controller, updateWeights}: WeightExplorerProps) {
  useEffect(() => {

  }, [controller]);


}
