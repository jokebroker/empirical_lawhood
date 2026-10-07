-- Scientific figure assets remain unchanged; use their PDF wrappers in LaTeX.
function Image(img)
  if FORMAT:match('latex') and img.src:match('assets/figure%d%.png$') then
    img.src = img.src:gsub('%.png$', '.pdf')
  end
  return img
end
local function latex(blocks)
  return pandoc.write(pandoc.Pandoc(blocks),'latex'):gsub('%s+$','')
end
local table_count = 0
function Table(tbl)
  table_count = table_count + 1
  if #tbl.colspecs == 3 then
    tbl.colspecs[1][2] = 0.25
    tbl.colspecs[2][2] = 0.34
    tbl.colspecs[3][2] = 0.41
  elseif #tbl.colspecs == 4 then
    tbl.colspecs[1][2] = 0.28
    tbl.colspecs[2][2] = 0.24
    tbl.colspecs[3][2] = 0.24
    tbl.colspecs[4][2] = 0.24
  end
  -- Captioned main-text tables stay at their point of discussion.
  if FORMAT:match('latex') and #tbl.caption.long > 0 then
    local cols={}
    for _,spec in ipairs(tbl.colspecs) do
      table.insert(cols,'>{\\raggedright\\arraybackslash}p{\\dimexpr '..tostring(spec[2])..'\\linewidth-2\\tabcolsep\\relax}')
    end
    local out={'\\begin{table}[H]','\\centering','\\fontsize{10}{12}\\selectfont','\\renewcommand{\\arraystretch}{1.1}','\\begin{tabular}{@{}'..table.concat(cols)..'@{}}','\\toprule'}
    local function row(r,bold)
      local cells={}
      for _,c in ipairs(r.cells) do
        local v=latex(c.contents)
        if bold then v='\\textbf{'..v..'}' end
        table.insert(cells,v)
      end
      return table.concat(cells,' & ')..' \\\\'
    end
    for _,r in ipairs(tbl.head.rows) do table.insert(out,row(r,true)) end
    table.insert(out,'\\midrule')
    for _,body in ipairs(tbl.bodies) do
      for _,r in ipairs(body.body) do table.insert(out,row(r,false));table.insert(out,'\\addlinespace[4pt]') end
    end
    table.insert(out,'\\bottomrule');table.insert(out,'\\end{tabular}')
    table.insert(out,'\\caption{'..latex(tbl.caption.long)..'}')
    table.insert(out,'\\end{table}')
    return pandoc.RawBlock('latex',table.concat(out,'\n'))
  end
  return tbl
end
local compact_refs = false
function Header(h)
  if FORMAT:match('latex') and h.identifier == 'references' then
    compact_refs = true
    return {pandoc.RawBlock('latex','\\FloatBarrier\n\\begingroup\\fontsize{9.5}{11.3}\\selectfont\\setlength{\\parskip}{3pt}'),h}
  end
  if FORMAT:match('latex') and h.level == 1 then
    return {pandoc.RawBlock('latex','\\FloatBarrier\n\\Needspace{8\\baselineskip}'),h}
  end
  return h
end
function Pandoc(doc)
  if FORMAT:match('latex') then
    for i=2,#doc.blocks do
      local b=doc.blocks[i]
      local prev=doc.blocks[i-1]
      if b.t == 'Para' and #b.content == 1 and b.content[1].t == 'Math' and b.content[1].mathtype == 'DisplayMath' and prev.t == 'Para' then
        prev.content:insert(pandoc.RawInline('latex','\\nopagebreak[4]'))
      end
    end
  end
  if compact_refs then doc.blocks:insert(pandoc.RawBlock('latex','\\endgroup')) end
  return doc
end
