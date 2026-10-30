// Visual design directions. Each non-default design is a CSS layer scoped to
// :root[data-design=<id>] and works in both light and dark themes.
import './cupertino.css';
import './pro.css';
import './glass.css';
import './keynote.css';

export const designs=[
  {id:'bench',name:'Calibration bench',description:'The original warm graphite workbench.',swatches:['#171716','#ff7a3d','#f1efea']},
  {id:'cupertino',name:'Cupertino',description:'Native macOS app: vibrancy sidebar, grouped forms.',swatches:['#f5f5f7','#0071e3','#1d1d1f']},
  {id:'pro',name:'Pro Studio',description:'Pro-app graphite with the viewport as the hero.',swatches:['#1c1c1e','#ffd60a','#636366']},
  {id:'glass',name:'Liquid Glass',description:'Floating frosted panels over an ambient field.',swatches:['#c9d6ff','#5e5ce6','#ffffff']},
  {id:'keynote',name:'Keynote',description:'Editorial product-page type and generous space.',swatches:['#ffffff','#000000','#bf5af2']},
] as const;
export type DesignId=typeof designs[number]['id'];
export const storedDesign=():DesignId=>{try{const v=localStorage.getItem('studio-design');return designs.some(d=>d.id===v)?v as DesignId:'cupertino';}catch{return 'cupertino';}};
