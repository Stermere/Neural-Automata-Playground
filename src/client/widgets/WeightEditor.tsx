import React, { useState } from 'react';
import styles from './weightEditor.module.css';

type Weights3D = number[][][][];

interface WeightEditorProps {
  initialWeights: Weights3D;
  onChange?: (weights: Weights3D) => void;
}

export default function WeightEditor({ initialWeights, onChange }: WeightEditorProps) {
  return null;
}