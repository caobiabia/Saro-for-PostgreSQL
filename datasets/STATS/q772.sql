select  count(*) from comments as c,  		posts as p,           votes as v,  		badges as b,          users as u  where u.Id =c.UserId 	and c.UserId = p.OwnerUserId  	and p.OwnerUserId = v.UserId 	and v.UserId = b.UserId  AND c.CreationDate>='2010-07-22 00:39:46'::timestamp  AND p.CommentCount>=0  AND p.FavoriteCount<=3  AND u.Reputation>=1  AND u.Reputation<=1068  AND u.UpVotes>=0  AND u.CreationDate>='2010-08-26 19:28:33'::timestamp  AND u.CreationDate<='2014-08-25 16:11:33'::timestamp  AND v.BountyAmount>=0  AND v.CreationDate>='2010-07-19 00:00:00'::timestamp;