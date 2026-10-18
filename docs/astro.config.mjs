import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://snoozeml.github.io',
  base: '/SnoozeML',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    mermaid(),
    starlight({
      title: 'SnoozeML',
      description: 'Single-binary, scale-to-zero ML inference daemon in Rust',
      social: {
        github: 'https://github.com/SnoozeML/SnoozeML',
      },
      customCss: [
        'katex/dist/katex.min.css',
      ],
      sidebar: [
        { label: 'Introduction', slug: 'introduction' },
        {
          label: 'Getting Started',
          items: [
            'getting-started/installation',
            'getting-started/quickstart',
            'getting-started/downloading-models',
          ],
        },
        {
          label: 'Serving Models',
          items: [
            'serving/starting-the-server',
            'serving/configuration',
            'serving/hardware-acceleration',
            'serving/supported-formats',
            'serving/custom-models',
          ],
        },
        {
          label: 'Scaling & Performance Mechanics',
          items: [
            'scaling/overview',
            'scaling/dynamic-batching',
            'scaling/scale-to-zero',
            'scaling/horizontal-autoscaling',
            'scaling/memory-and-oom',
          ],
        },
        {
          label: 'Authentication & Security',
          items: [
            'security/overview',
            'security/api-keys',
            'security/jwt',
            'security/endpoint-security',
          ],
        },
        {
          label: 'Client Usage & Inference',
          items: [
            'client/python-sdk',
            {
              label: 'Python SDK API Reference (Sphinx) ↗',
              link: '/SnoozeML/python/',
              attrs: { target: '_blank' },
            },
            'client/inference-requests',
            'client/grpc-inference',
            'client/tensors',
            'client/errors',
          ],
        },
        {
          label: 'Event Streaming & Message Brokers',
          items: [
            'streaming/quickstart',
            'streaming/overview',
            'streaming/configuration',
          ],
        },
        {
          label: 'Operations & Monitoring',
          items: [
            'operations/tui-monitor',
            'operations/model-lifecycle',
            'operations/health-probes',
            'operations/prometheus-metrics',
            'operations/production',
          ],
        },
        {
          label: 'Reference & Architecture',
          items: [
            'reference/architecture',
            'reference/cli-reference',
            'reference/api-reference',
            'reference/grpc-api-reference',
          ],
        },
      ],
    }),
  ],
});
