Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression

$out = Join-Path $PSScriptRoot 'Carbon-Eye_Project_Presentation_Google-Slides-Compatible.pptx'
$tmp = Join-Path $env:TEMP ('carbon-eye-ppt-' + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

function Put-Text([string]$path, [string]$content) {
  $full = Join-Path $tmp $path
  New-Item -ItemType Directory -Path (Split-Path $full) -Force | Out-Null
  [System.IO.File]::WriteAllText($full, $content, [System.Text.UTF8Encoding]::new($false))
}
function Esc([string]$s) { [System.Security.SecurityElement]::Escape($s) }
function Shape([int]$id, [int]$x, [int]$y, [int]$cx, [int]$cy, [string]$text, [int]$size, [string]$color, [bool]$bold=$false, [string]$fill='') {
  $body = ''
  foreach ($line in ($text -split "`n")) {
    $b = if ($bold) {' b="1"'} else {''}
    $body += "<a:p><a:r><a:rPr lang=`"en-US`" sz=`"$size`"$b><a:solidFill><a:srgbClr val=`"$color`"/></a:solidFill></a:rPr><a:t>$(Esc $line)</a:t></a:r><a:endParaRPr lang=`"en-US`" sz=`"$size`"/></a:p>"
  }
  $fillXml = if ($fill) { "<a:solidFill><a:srgbClr val=`"$fill`"/></a:solidFill>" } else { '<a:noFill/>' }
  "<p:sp><p:nvSpPr><p:cNvPr id=`"$id`" name=`"Text $id`"/><p:cNvSpPr txBox=`"1`"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=`"$x`" y=`"$y`"/><a:ext cx=`"$cx`" cy=`"$cy`"/></a:xfrm><a:prstGeom prst=`"rect`"><a:avLst/></a:prstGeom>$fillXml<a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap=`"square`"/><a:lstStyle/>$body</p:txBody></p:sp>"
}

Put-Text '[Content_Types].xml' @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
'@
for ($i=1; $i -le 10; $i++) { Add-Content -LiteralPath (Join-Path $tmp '[Content_Types].xml') "<Override PartName=`"/ppt/slides/slide$i.xml`" ContentType=`"application/vnd.openxmlformats-officedocument.presentationml.slide+xml`"/>" }
Add-Content -LiteralPath (Join-Path $tmp '[Content_Types].xml') '</Types>'

Put-Text '_rels/.rels' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>'
Put-Text 'ppt/presentation.xml' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>'
for ($i=1; $i -le 10; $i++) { Add-Content (Join-Path $tmp 'ppt/presentation.xml') "<p:sldId id=`"$(255+$i)`" r:id=`"rId$($i+1)`"/>" }
Add-Content (Join-Path $tmp 'ppt/presentation.xml') '</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
Put-Text 'ppt/_rels/presentation.xml.rels' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
for ($i=1; $i -le 10; $i++) { Add-Content (Join-Path $tmp 'ppt/_rels/presentation.xml.rels') "<Relationship Id=`"rId$($i+1)`" Type=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide`" Target=`"slides/slide$i.xml`"/>" }
Add-Content (Join-Path $tmp 'ppt/_rels/presentation.xml.rels') '</Relationships>'

Put-Text 'ppt/theme/theme1.xml' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Carbon-Eye"><a:themeElements><a:clrScheme name="Carbon"><a:dk1><a:srgbClr val="16352B"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="276749"/></a:dk2><a:lt2><a:srgbClr val="EAF4EE"/></a:lt2><a:accent1><a:srgbClr val="2F855A"/></a:accent1><a:accent2><a:srgbClr val="D69E2E"/></a:accent2><a:accent3><a:srgbClr val="4299E1"/></a:accent3><a:accent4><a:srgbClr val="9F7AEA"/></a:accent4><a:accent5><a:srgbClr val="ED8936"/></a:accent5><a:accent6><a:srgbClr val="38B2AC"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Carbon"><a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Carbon"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="accent1"/></a:solidFill><a:noFill/></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:noFill/></a:ln><a:ln w="25400"><a:noFill/></a:ln><a:ln w="38100"><a:noFill/></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="lt1"/></a:solidFill><a:noFill/></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'
Put-Text 'ppt/slideMasters/slideMaster1.xml' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld name="Carbon-Eye"><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F7FAF7"/></a:solidFill></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483648" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>'
Put-Text 'ppt/slideMasters/_rels/slideMaster1.xml.rels' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>'
Put-Text 'ppt/slideLayouts/slideLayout1.xml' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld></p:sldLayout>'
Put-Text 'ppt/slideLayouts/_rels/slideLayout1.xml.rels' '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>'

$slides = @(
  @('Carbon-Eye', 'Carbon Stock & Above-Ground Biomass Estimation`nA geospatial decision-support system for selected Kenyan counties', 'Project presentation'),
  @('What is Carbon-Eye?', 'Carbon-Eye is a Streamlit web application that estimates above-ground biomass (AGB) and carbon stock across selected Kenyan counties.`n`nIt combines satellite observations, environmental data and machine-learning models in Google Earth Engine to turn complex spatial data into interactive maps and county-level summaries.', 'Purpose: make carbon and biomass screening more accessible, transparent and usable.'),
  @('Why estimate carbon and biomass?', 'Forests and vegetation store carbon in their biomass. Measuring their distribution helps us understand ecosystem condition and change.`n`nCarbon-Eye supports: climate-action planning; restoration screening; landscape monitoring; county prioritisation; and evidence-based conservation discussions.', 'Important boundary: results are model estimates, not field inventories, verified carbon credits, or certification decisions.'),
  @('How Carbon-Eye works', '1. Select Kenyan county/counties and an ESA CCI AGB reference year.`n2. Google Earth Engine builds a cloud-filtered, multi-source predictor stack.`n3. The app samples reference carbon and splits data into training and testing sets.`n4. Random Forest, Gradient Tree Boosting and SVM learn carbon patterns.`n5. Carbon-Eye maps estimates, checks model performance and summarises counties.', 'Inputs to predictor stack to model training to validation to maps, statistics and downloads'),
  @('Who can use it?', 'County environment and forestry teams`nConservation and restoration organisations`nResearchers and students`nCarbon-project teams conducting early screening`nLand-use planners and NGOs', 'Users need an Earth Engine-enabled Google account and a Google Cloud project. The interface is designed for non-specialists as well as technical users.'),
  @('Data sources used', 'Reference target: ESA CCI Above-Ground Biomass V6.0`nOptical imagery: Sentinel-2 Surface Reflectance (Copernicus)`nRadar: Sentinel-1 GRD and JAXA ALOS PALSAR`nLand cover: Google Dynamic World`nTerrain: SRTM DEM`nClimate: WorldClim BIO`nSoils: OpenLandMap Soil Organic Carbon`nStructure & temperature: Meta Canopy Height and MODIS LST`nBoundaries: geoBoundaries ADM1', 'All sources are accessed and processed through Google Earth Engine.'),
  @('Features used by the model', 'Vegetation greenness & moisture: Sentinel-2 bands, NDVI, EVI, SAVI, NDMI and NDRE`nVegetation structure: Sentinel-1 VH radar, SAR texture (contrast), PALSAR HH/HV, canopy height`nSite conditions: elevation, slope, aspect, temperature, rainfall, soil organic carbon and land-surface temperature', 'These predictors capture vegetation condition, structure and the environmental conditions that influence biomass.'),
  @('Model design and carbon estimate', 'Training target: ESA AGB reference layer converted to carbon stock using 0.47 Mg C per Mg dry biomass (IPCC default).`n`nModels: Random Forest, Gradient Tree Boosting and Support Vector Machine (SVM).`n`nEnsemble: average of the three predicted carbon maps, reducing reliance on one modelling approach.', 'Outputs are reported as tonnes of carbon per hectare (t C/ha); the app can also display AGB.'),
  @('Key system features', 'Interactive map: select model, switch Carbon/AGB and inspect locations.`nModel comparison: RF-versus-GTB difference and three-model spread (uncertainty).`nValidation: held-out test data with RMSE, MAE, R² and actual-versus-predicted charts.`nCounty insights: zonal mean, min, max and total carbon statistics; downloadable CSV.`nFeature importance: identifies influential variables for RF and GTB.`nLearning guide and reporting/export tools.', 'The app caches matching runs and computes heavier diagnostics on demand to improve usability.'),
  @('Responsible interpretation & next steps', 'Use Carbon-Eye to screen, compare and target areas, not to replace field plots or verification.`n`nBefore any investment, crediting or certification decision: validate with local field data; assess uncertainty; check land tenure, baselines, leakage and permanence; and follow the relevant methodology.', 'Thank you`nCarbon-Eye: seeing Kenya landscape carbon more clearly')
)
for ($i=0; $i -lt $slides.Count; $i++) {
  $n=$i+1; $s=$slides[$i]
  $content = Shape 2 650000 360000 10900000 760000 $s[0] 2800 '276749' $true
  $content += Shape 3 760000 1450000 10600000 3400000 $s[1] 1650 '16352B' $false
  $content += Shape 4 760000 5500000 10600000 650000 $s[2] 1150 'FFFFFF' $false '2F855A'
  $xml = "<?xml version=`"1.0`" encoding=`"UTF-8`" standalone=`"yes`"?><p:sld xmlns:a=`"http://schemas.openxmlformats.org/drawingml/2006/main`" xmlns:r=`"http://schemas.openxmlformats.org/officeDocument/2006/relationships`" xmlns:p=`"http://schemas.openxmlformats.org/presentationml/2006/main`"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id=`"1`" name=`"`"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>$content</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
  Put-Text "ppt/slides/slide$n.xml" $xml
  Put-Text "ppt/slides/_rels/slide$n.xml.rels" '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>'
}

if (Test-Path $out) { Remove-Item -LiteralPath $out -Force }
$archive = [System.IO.Compression.ZipFile]::Open($out, [System.IO.Compression.ZipArchiveMode]::Create)
try {
  Get-ChildItem -LiteralPath $tmp -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($tmp.Length).TrimStart('\').Replace('\', '/')
    $entry = $archive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
    $input = [System.IO.File]::OpenRead($_.FullName)
    $output = $entry.Open()
    try { $input.CopyTo($output) } finally { $output.Dispose(); $input.Dispose() }
  }
} finally { $archive.Dispose() }
Remove-Item -LiteralPath $tmp -Recurse -Force
Write-Output "Created $out"
