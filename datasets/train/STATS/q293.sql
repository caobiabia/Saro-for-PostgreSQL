select  count(*) from comments as c,  		posts as p,           votes as v,  		badges as b,          users as u  where u.Id =c.UserId 	and c.UserId = p.OwnerUserId  	and p.OwnerUserId = v.UserId 	and v.UserId = b.UserId  AND p.CreationDate<='2014-08-31 08:40:26'::timestamp  AND u.Views>=0  AND u.Views<=273  AND u.DownVotes=0  AND v.VoteTypeId=2  AND v.CreationDate='2012-08-30 00:00:00'::timestamp;