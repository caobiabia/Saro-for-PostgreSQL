select  count(*) from comments as c,  		postHistory as ph where c.UserId = ph.UserId  AND c.Score=0  AND c.CreationDate<='2014-09-13 20:19:19'::timestamp  AND ph.PostHistoryTypeId=1  AND ph.CreationDate>='2010-07-31 14:29:15'::timestamp  AND ph.CreationDate<='2014-08-21 18:14:18'::timestamp;