import { useState } from 'react';
import { App, Button, Input, Space } from 'antd';
import { FolderOpenOutlined } from '@ant-design/icons';
import { api } from '../api/client';

interface Props {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label: string;
  kind?: 'file' | 'directory' | 'model';
}

export default function PathInput({ value, onChange, placeholder, label, kind = 'directory' }: Props) {
  const { message } = App.useApp();
  const [picking, setPicking] = useState(false);
  const pick = async (type: 'file' | 'directory') => {
    setPicking(true);
    try {
      const result = await api.pickPath(type);
      if (result.path) onChange(result.path);
    } catch (error) {
      message.error((error as Error).message);
    } finally {
      setPicking(false);
    }
  };
  return <>
    <Input data-ai-ignore value={value} aria-label={label} placeholder={placeholder} allowClear onChange={(e) => onChange(e.target.value)} />
    <Space className="path-actions">
      <Button size="small" icon={<FolderOpenOutlined />} loading={picking} onClick={() => pick(kind === 'file' ? 'file' : 'directory')}>
        {kind === 'file' ? 'Select file' : 'Select folder'}
      </Button>
      {kind === 'model' && <Button size="small" disabled={picking} onClick={() => pick('file')}>Select GGUF file</Button>}
      <span className="muted">{kind === 'file' ? 'Select a file, or enter the file path manually' : kind === 'model' ? 'Select a model directory / GGUF file, or enter a path / repository link' : 'Select a folder, or enter the directory path manually'}</span>
    </Space>
  </>;
}
