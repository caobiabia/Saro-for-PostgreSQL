select  count(*) from comments as c,  		postHistory as ph where c.UserId = ph.UserId  AND c.Score=0  AND ph.PostHistoryTypeId=3;
