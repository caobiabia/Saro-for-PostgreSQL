select  count(*) from comments as c,  		posts as p,  		postHistory as ph,          votes as v,          users as u  where u.Id = c.UserId 	and c.UserId = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = v.UserId  AND c.CreationDate>='2010-08-17 18:42:30'::timestamp  AND ph.PostHistoryTypeId=2  AND ph.CreationDate>='2010-07-21 14:00:03'::timestamp  AND ph.CreationDate<='2014-06-19 21:48:47'::timestamp  AND p.Score>=-1  AND p.ViewCount=0  AND p.CreationDate>='2010-09-21 00:47:15'::timestamp  AND u.Views>=0  AND u.DownVotes>=0  AND u.UpVotes<=40  AND v.CreationDate>='2010-07-26 00:00:00'::timestamp;