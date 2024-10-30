select  count(*) from comments as c,  		postHistory as ph where c.UserId = ph.UserId  AND ph.PostHistoryTypeId=5;
