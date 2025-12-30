// Example integration of ChatModal with Docusaurus layout
// This would typically go in src/theme/Layout/index.js or a similar layout component

import React from 'react';
import Layout from '@theme-original/Layout';
import ChatModal from '../components/ChatModal';

export default function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props}>
        {props.children}
      </Layout>
      {/* Add the chat modal to every page */}
      <ChatModal pageContext={props.location?.pathname || ''} />
    </>
  );
}