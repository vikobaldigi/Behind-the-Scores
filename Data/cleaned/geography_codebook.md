# Geography Codebook

## Introduction
This codebook documents the Geography codebook. It covers coordinates, addresses, and civic
geography (borough, district, ZIP, etc.) derived from school records and external sources.

## Variable Overview
| variable             | label                | type        | measurement_level   | unit   |   n_unique |   missing_pct | summary                                                                                                                   |
|:---------------------|:---------------------|:------------|:--------------------|:-------|-----------:|--------------:|:--------------------------------------------------------------------------------------------------------------------------|
| DBN                  | DBN                  | text        | free-text           |        |        410 |             0 | 71M519 (19, 3.7%); 76K698 (16, 3.1%); 76K673 (16, 3.1%); 76K529 (16, 3.1%); 72X377 (16, 3.1%); 76R600 (14, 2.7%); 71M459… |
| Location_Code        | Location Code        | text        | free-text           |        |        410 |             0 | M519 (19, 3.7%); K698 (16, 3.1%); K673 (16, 3.1%); K529 (16, 3.1%); X377 (16, 3.1%); R600 (14, 2.7%); M459 (6, 1.2%); M5… |
| School_Name          | School Name          | text        | free-text           |        |        410 |             0 | TALENT UNLIMITED HIGH SCHOOL (19, 3.7%); SOUTH BROOKLYN COMMUNITY HIGH SCHOOL (16, 3.1%); EAST BROOKLYN COMMUNITY HIGH S… |
| latitude             | Latitude             | datetime    | interval            |        |        230 |             0 | 40.677968 (64, 12.5%); 40.765638 (31, 6.0%); 40.64282 (14, 2.7%); 40.743407 (7, 1.4%); 40.693002 (6, 1.2%); 40.839775 (6… |
| longitude            | Longitude            | datetime    | interval            |        |        229 |             0 | -74.014363 (64, 12.5%); -73.959777 (31, 6.0%); -74.079263 (14, 2.7%); -74.00254 (7, 1.4%); -73.869305 (6, 1.2%); -73.839… |
| formatted_address    | Formatted Address    | text        | free-text           |        |        278 |             0 | 173 CONOVER STREET (64, 12.5%); 351 WEST 18 STREET (7, 1.4%); 925 ASTOR AVENUE (6, 1.2%); 500 EAST FORDHAM ROAD (6, 1.2%… |
| ZIP Code             | ZIP Code             | datetime    | interval            |        |        127 |             0 | 10065 (31, 6.0%); 10301 (15, 2.9%); 10457 (13, 2.5%); 11236 (12, 2.3%); 10011 (11, 2.1%); 10002 (11, 2.1%); 10456 (10, 1… |
| Neighborhood         | Neighborhood         | text        | free-text           |        |         43 |             0 | Downtown - Heights - Park Slope (82, 16.0%); Upper East Side (33, 6.4%); Fordham - Bronx Park (23, 4.5%); Chelsea - Clin… |
| Borough_Code_Derived | Borough Code Derived | categorical | nominal             |        |          5 |             0 | K (159, 31.0%); M (126, 24.6%); X (123, 24.0%); Q (80, 15.6%); R (25, 4.9%)                                               |
| Borough_Derived      | Borough Derived      | categorical | nominal             |        |          5 |             0 | Brooklyn (159, 31.0%); Manhattan (126, 24.6%); Bronx (123, 24.0%); Queens (80, 15.6%); Staten Island (25, 4.9%)           |
| District_Derived     | District Derived     | datetime    | interval            |        |         21 |             0 | 71 (118, 23.0%); 72 (114, 22.2%); 76 (105, 20.5%); 73 (76, 14.8%); 77 (74, 14.4%); 2 (4, 0.8%); 7 (2, 0.4%); 10 (2, 0.4%… |
