select  count(*) from comments as c,          postHistory as ph,  		badges as b,          users as u  where u.Id = c.UserId 	and u.Id = ph.UserId 	and u.Id = b.UserId  AND c.Score=0  AND c.CreationDate>='2010-07-20 14:17:12'::timestamp  AND ph.PostHistoryTypeId=1  AND u.Reputation>=1  AND u.Reputation<=358  AND u.Views>=0;