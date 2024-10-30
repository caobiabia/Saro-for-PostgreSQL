select  count(*) from comments as c,  		postHistory as ph where c.UserId = ph.UserId  AND c.CreationDate<='2014-09-07 22:15:01'::timestamp  AND ph.PostHistoryTypeId=1;
