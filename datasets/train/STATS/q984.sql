select  count(*) from comments as c,  		posts as p,          users as u where c.UserId = u.Id 	and u.Id = p.OwnerUserId  AND p.CommentCount<=15  AND p.FavoriteCount<=14  AND p.CreationDate>='2010-07-25 18:02:06'::timestamp  AND p.CreationDate<='2014-09-08 23:43:41'::timestamp  AND u.Reputation<=373  AND u.UpVotes>=0;