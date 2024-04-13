SELECT
    set1.Sample_ID,
    set1.Site_ID,
    set1.Reef_ID,
    set1.Date_Day,
    set1.Date_Month,
    set1.Date_Year,
    set1.Depth_m,
    sit.Latitude_Degrees,
    sit.Longitude_Degrees,
    sit.Distance_to_Shore,
    el.Exposure,
    sit.Turbidity,
    sit.Cyclone_Frequency,
    et.ClimSST,
	et.Temperature_Kelvin,
	et.Temperature_Mean,
	et.Temperature_Minimum,
	et.Temperature_Maximum,
	et.Temperature_Kelvin_Standard_Deviation,
	et.Windspeed,
	et.SSTA,
	et.SSTA_Standard_Deviation,
	et.SSTA_Mean,
	et.SSTA_Minimum,
	et.SSTA_Maximum,
	et.SSTA_Frequency,
	et.SSTA_Frequency_Standard_Deviation,
	et.SSTA_FrequencyMax,
	et.SSTA_FrequencyMean,
	et.SSTA_DHW,
	et.SSTA_DHW_Standard_Deviation,
	et.SSTA_DHWMax,
	et.SSTA_DHWMean,
	et.TSA,
	et.TSA_Standard_Deviation,
	et.TSA_Minimum,
	et.TSA_Maximum,
	et.TSA_Mean,
	et.TSA_Frequency,
	et.TSA_Frequency_Standard_Deviation,
	et.TSA_FrequencyMax,
	et.TSA_FrequencyMean,
	et.TSA_DHW,
	et.TSA_DHW_Standard_Deviation,
	et.TSA_DHWMax,
	et.TSA_DHWMean,
    bll.Bleaching_Level,
    bt.Percent_Bleached,
    bt.Percent_Bleaching_Old_Method,
    bt.S1, bt.S2, bt.S3, bt.S4,
    bpsl.Bleaching_Prevalence_Score,
    bt.Bleaching_Prevalence_Score as "Bleaching_Prevalence_Score_ID",
    scl.Severity_Code,
    bt.Severity_Code as "Severity_ID",
    bt.bleach_intensity,
    bt.Number_Bleached_Colonies,
    ct.Percent_Hard_Coral,
    ct.Percent_Macroalgae,
    stl.Substrate_Name,
    onl.Ocean_Name,
    rnl.Realm_Name,
    enl.Ecoregion_Name,
    cnl.Country_Name,
    sipnl.State_Island_Province_Name,
    sit.Site_Name,
    ctnl1.City_Town_Name as "City_Town_Name_1",
    ctnl2.City_Town_Name as "City_Town_Name_2",
    ctnl3.City_Town_Name as "City_Town_Name_3",
    ctnl4.City_Town_Name as "City_Town_Name_4",
    dsl.Data_Source,
    dsl.Sample_Method,
    set1.Comments as "Sample_Comments",
    sit.Comments as "Site Comments",
    ct.Comments as "Cover Comments"
FROM Sample_Event_tbl set1
LEFT JOIN Site_Info_tbl sit ON set1.Site_ID = sit.Site_ID
LEFT JOIN Ocean_Name_LUT onl ON sit.Ocean_Name = onl.Ocean_ID
LEFT JOIN Realm_Name_LUT rnl ON sit.Realm_Name = rnl.Realm_ID
LEFT JOIN Ecoregion_Name_LUT enl ON sit.Ecoregion_Name = enl.Ecoregion_ID
LEFT JOIN Country_Name_LUT cnl ON sit.Country_Name = cnl.Country_ID
LEFT JOIN State_Island_Province_Name_LUT sipnl ON sit.State_Island_Province_Name = sipnl.State_Island_Province_ID
LEFT JOIN City_Town_Name_LUT ctnl1 ON sit.City_Town_Name = ctnl1.City_Town_ID
LEFT JOIN City_Town_Name_LUT ctnl2 ON sit.City_Town_Name_2 = ctnl2.City_Town_ID
LEFT JOIN City_Town_Name_LUT ctnl3 ON sit.City_Town_Name_3 = ctnl3.City_Town_ID
LEFT JOIN City_Town_Name_LUT ctnl4 ON sit.City_Town_Name_4 = ctnl4.City_Town_ID
LEFT JOIN Exposure_LUT el ON sit.Exposure = el.Exposure_ID
LEFT JOIN Data_Source_LUT dsl ON sit.Data_Source = dsl.Data_Source_ID
LEFT JOIN Cover_tbl ct ON set1.Sample_ID = ct.Sample_ID
LEFT JOIN Substrate_Type_LUT stl ON ct.Substrate_Type = stl.Substrate_ID
LEFT JOIN Bleaching_tbl bt ON set1.Sample_ID = bt.Sample_ID
LEFT JOIN Bleaching_Level_LUT bll ON bt.Bleaching_Level = bll.Bleaching_Level_ID
LEFT JOIN Bleaching_Prevalence_Score_LUT bpsl ON bt.Bleaching_Prevalence_Score = bpsl.Bleaching_Prevalence_ID
LEFT JOIN Severity_Code_LUT scl ON bt.Severity_Code = scl.Severity_ID
LEFT JOIN Environmental_tbl et ON set1.Sample_ID = et.Sample_ID
