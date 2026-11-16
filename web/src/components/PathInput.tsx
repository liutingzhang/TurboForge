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
        {kind === 'file' ? '选择文件' : '选择文件夹'}
      </Button>
      {kind === 'model' && <Button size="small" disabled={picking} onClick={() => pick('file')}>选择 GGUF 文件</Button>}
      <span className="muted">{kind === 'file' ? '选择文件，或手动填写文件路径' : kind === 'model' ? '选择模型目录 / GGUF 文件，或填写路径 / 仓库链接' : '选择文件夹，或手动填写目录路径'}</span>
    </Space>
  </>;
}
