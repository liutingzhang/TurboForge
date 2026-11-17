import { useEffect, useState } from 'react';
import type { SelectionBlock } from '../api/assistant';
import { areaSelection, blockFrom, readable, selectionRoot } from '../utils/assistantContext';

export function usePageSelection(active: boolean, onSelect: (blocks: SelectionBlock[]) => void, onError: (message: string) => void, onCancel: () => void) {
  const [rectangle, setRectangle] = useState<DOMRect | null>(null);
  useEffect(() => {
    if (!active) { setRectangle(null); return; }
    let start: {x: number; y: number} | null = null;
    const previousCursor = document.body.style.cursor;
    document.body.style.cursor = 'crosshair';
    const suppress = (event: Event) => {
      if (event.target instanceof Element && event.target.closest('[data-ai-selection-control]')) return;
      event.preventDefault(); event.stopImmediatePropagation();
    };
    const down = (event: PointerEvent) => {
      if (event.target instanceof Element && event.target.closest('[data-ai-selection-control]')) return;
      suppress(event);
      if (event.button === 0) start = {x: event.clientX, y: event.clientY};
    };
    const move = (event: PointerEvent) => {
      if (start) {
        suppress(event);
        setRectangle(new DOMRect(Math.min(start.x, event.clientX), Math.min(start.y, event.clientY), Math.abs(start.x - event.clientX), Math.abs(start.y - event.clientY)));
      } else {
        const element = document.elementFromPoint(event.clientX, event.clientY);
        setRectangle(element && readable(element) ? selectionRoot(element).getBoundingClientRect() : null);
      }
    };
    const up = (event: PointerEvent) => {
      if (!start) return;
      suppress(event);
      // React can unmount the selection listener after pointerup but before the
      // browser's synthetic click. Keep that click captured through the handoff.
      const swallowClick = (click: Event) => {click.preventDefault(); click.stopImmediatePropagation();};
      window.addEventListener('click', swallowClick, {capture:true,once:true});
      setTimeout(() => window.removeEventListener('click', swallowClick, true), 250);
      try {
        let blocks: SelectionBlock[];
        if (Math.abs(event.clientX - start.x) + Math.abs(event.clientY - start.y) > 8) {
          blocks = areaSelection(new DOMRect(Math.min(start.x, event.clientX), Math.min(start.y, event.clientY), Math.abs(start.x - event.clientX), Math.abs(start.y - event.clientY)));
        } else {
          const element = document.elementFromPoint(event.clientX, event.clientY);
          const block = element ? blockFrom(element) : null;
          blocks = block ? [block] : [];
        }
        if (!blocks.length) onError('No readable text here; for images or cross-origin content, select the description instead or type it yourself');
        else onSelect(blocks);
      } catch (error) { onError((error as Error).message); }
      start = null;
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { suppress(event); onCancel(); }
      else if (event.key === 'Enter' || event.key === ' ') suppress(event);
    };
    window.addEventListener('pointerdown', down, true);
    window.addEventListener('pointermove', move, true);
    window.addEventListener('pointerup', up, true);
    window.addEventListener('click', suppress, true);
    window.addEventListener('mousedown', suppress, true);
    window.addEventListener('mouseup', suppress, true);
    window.addEventListener('keydown', key, true);
    return () => {
      document.body.style.cursor = previousCursor;
      window.removeEventListener('pointerdown', down, true); window.removeEventListener('pointermove', move, true);
      window.removeEventListener('pointerup', up, true); window.removeEventListener('click', suppress, true);
      window.removeEventListener('mousedown', suppress, true); window.removeEventListener('mouseup', suppress, true);
      window.removeEventListener('keydown', key, true);
    };
  }, [active, onSelect, onError, onCancel]);
  return rectangle;
}
