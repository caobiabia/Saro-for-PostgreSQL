select  count(*) from comments as c,          postHistory as ph,  		badges as b,          users as u  where u.Id = c.UserId 	and u.Id = ph.UserId 	and u.Id = b.UserId  AND ph.PostHistoryTypeId=25;
