import { theme, type ThemeConfig } from 'antd';

// Minimal light "precision instrument" palette: warm paper, ink type, a single
// ember accent. Applied through Ant Design tokens so every component shares it.
export const forgeTheme: ThemeConfig = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: '#d9480f',
    colorInfo: '#1971c2',
    colorSuccess: '#2f9e44',
    colorWarning: '#e8890c',
    colorError: '#d6336c',
    colorBgBase: '#f5f3ef',
    colorBgContainer: '#ffffff',
    colorBgElevated: '#ffffff',
    colorBgLayout: '#f5f3ef',
    colorBorder: '#e6e2da',
    colorBorderSecondary: '#efece6',
    colorText: '#1b1e22',
    colorTextSecondary: '#5b636d',
    colorTextTertiary: '#9aa1ab',
    borderRadius: 8,
    fontSize: 13,
    fontFamily:
      'system-ui, "Segoe UI", "Microsoft YaHei", "PingFang SC", Roboto, Helvetica, Arial, sans-serif',
    wireframe: false,
  },
  components: {
    Card: { colorBgContainer: '#ffffff', headerBg: 'transparent', colorBorderSecondary: '#efece6' },
    Select: { colorBgElevated: '#ffffff', optionSelectedBg: '#fdeee4', optionActiveBg: '#fbfaf8' },
    Table: { colorBgContainer: 'transparent', headerBg: '#fbfaf8', rowHoverBg: '#fbfaf8' },
    Steps: { colorPrimary: '#d9480f' },
    Layout: { headerBg: 'transparent', bodyBg: 'transparent' },
    Tooltip: { colorBgSpotlight: '#1b1e22' },
    Button: { primaryShadow: '0 3px 12px rgba(217,72,15,0.25)' },
  },
};
